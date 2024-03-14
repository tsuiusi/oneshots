interface Props {
  children: ReactNode;
  onClose: () => void;
}

const alert = ({ onClose, children }: Props) => {
  return (
    <div className="alert alert-primary alert-dismissable">
      { children }
      <button justify-content="space-between" type="button" className="close float-right" onClick={ onClose } data-dismiss="alert" aria-label="Close">
        <span aria-hidden="true">&times;</span>
      </button>
    </div>
  )
}

export default alert;